import argparse, json, ast
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

METRICS = [
    ("mean_ms", "Mean latency (ms)"),
    ("p95_ms", "p95 latency (ms)"),
    ("p99_ms", "p99 latency (ms)"),
    ("throughput_rps", "Throughput (req/s)"),
    ("sys_cpu_pct_mean", "CPU util mean (%)"),
    ("sys_rss_mb_peak", "Memory RSS peak (MB)"),
    ("hit_rate_warm", "Warm hit rate"),
    ("cost_saved_total", "Cost saved (total)"),
    # ("sys_gpu_util_pct_mean","GPU util mean (%)")  # include if you add GPU metrics
]

def _collect_run_dirs(paths):
    """
    Accept a mix of run dirs or parent dirs.
    Return all dirs that contain summary.csv (searches one level deep +
    handles the case where the provided path itself is a run dir).
    """
    out = []
    for p in paths:
        p = Path(p)
        if (p / "summary.csv").is_file():
            out.append(p)
        elif p.is_dir():
            # immediate subdirectories with summary.csv
            for sub in p.glob("*"):
                if sub.is_dir() and (sub / "summary.csv").is_file():
                    out.append(sub)
    # de-duplicate by resolved path
    seen, uniq = set(), []
    for d in out:
        rp = d.resolve()
        if rp not in seen:
            uniq.append(d)
            seen.add(rp)
    return uniq

def _policy_from_row(row: dict, rd: Path) -> str:
    """
    Prefer policy from the serialized 'cache' column (eviction),
    which may be a dict or a stringified dict. Fallback to folder name.
    """
    cache_col = row.get("cache")
    # direct dict
    if isinstance(cache_col, dict):
        ev = cache_col.get("eviction")
        if ev:
            return str(ev).upper()
    # stringified dict
    if isinstance(cache_col, str):
        try:
            maybe = ast.literal_eval(cache_col)
            if isinstance(maybe, dict):
                ev = maybe.get("eviction")
                if ev:
                    return str(ev).upper()
        except Exception:
            pass
    # folder-name heuristics
    name = rd.name.lower()
    if name.startswith("cost_aware"):
        return "COST_AWARE"
    if name.startswith("fifo"):
        return "FIFO"
    if name.startswith("lfu"):
        return "LFU"
    if name.startswith("lru"):
        return "LRU"
    # generic: token before first underscore
    return name.split("_", 1)[0].upper()

def load_runs(run_dirs):
    run_dirs = _collect_run_dirs(run_dirs)
    rows = []
    for rd in run_dirs:
        s = Path(rd) / "summary.csv"
        if s.is_file():
            df = pd.read_csv(s)
            if not df.empty:
                row = df.iloc[0].to_dict()
                row["policy"] = _policy_from_row(row, Path(rd))
                row["run_dir"] = Path(rd).as_posix()
                rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def plot_metric(df, key, title, outpng):
    plt.figure()
    sub = df[["policy", key]].copy()
    # order policies nicely if present
    order = [p for p in ["FIFO", "LFU", "LRU", "COST_AWARE"] if p in set(sub["policy"])]
    if order:
        sub["policy"] = pd.Categorical(sub["policy"], categories=order, ordered=True)
        sub = sub.sort_values("policy")
    sub = sub.set_index("policy")
    sub.plot(kind="bar", legend=False, ylabel=title, rot=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Paths to run dirs or a parent folder containing run dirs (each run must contain summary.csv)")
    ap.add_argument("--out", required=True, help="Output folder for PNGs and combined CSV")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_runs(args.runs)
    if df.empty:
        raise SystemExit("No valid summary.csv found in provided run dirs.")

    # save combined CSV
    combined_csv = out / "comparison_summary.csv"
    df.to_csv(combined_csv, index=False, encoding="utf-8")

    # plots
    for key, title in METRICS:
        if key in df.columns:
            plot_metric(df, key, title, out / f"{key}.png")

    # simple console print
    cols = ["policy"] + [k for k, _ in METRICS if k in df.columns]
    # stable presentation order like in plot
    if "policy" in df.columns:
        order = [p for p in ["FIFO", "LFU", "LRU", "COST_AWARE"] if p in set(df["policy"])]
        if order:
            df["policy"] = pd.Categorical(df["policy"], categories=order, ordered=True)
            df = df.sort_values("policy")
    print(df[cols].to_string(index=False))

if __name__ == "__main__":
    main()
